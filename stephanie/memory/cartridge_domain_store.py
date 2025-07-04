from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.models.cartridge_domain import CartridgeDomainORM


class CartridgeDomainStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "cartridge_domains"

    def insert(self, data: dict) -> CartridgeDomainORM:
        """
        Insert or update a domain classification entry.

        Expected dict keys: cartridge_id, domain, score
        """
        stmt = (
            pg_insert(CartridgeDomainORM)
            .values(**data)
            .on_conflict_do_nothing(index_elements=["cartridge_id", "domain"])
            .returning(CartridgeDomainORM.id)
        )

        result = self.session.execute(stmt)
        inserted_id = result.scalar()
        self.session.commit()

        if inserted_id and self.logger:
            self.logger.log("CartridgeDomainUpserted", data)

        return (
            self.session.query(CartridgeDomainORM)
            .filter_by(cartridge_id=data["cartridge_id"], domain=data["domain"])
            .first()
        )

    def get_domains(self, cartridge_id: int) -> list[CartridgeDomainORM]:
        return (
            self.session.query(CartridgeDomainORM)
            .filter_by(cartridge_id=cartridge_id)
            .order_by(CartridgeDomainORM.score.desc())
            .all()
        )

    def delete_domains(self, cartridge_id: int):
        self.session.query(CartridgeDomainORM).filter_by(cartridge_id=cartridge_id).delete()
        self.session.commit()
        if self.logger:
            self.logger.log("CartridgeDomainsDeleted", {"cartridge_id": cartridge_id})

    def set_domains(self, cartridge_id: int, domains: list[tuple[str, float]]):
        """
        Clear and re-add domains for the cartridge.
        :param domains: list of (domain, score) pairs
        """
        self.delete_domains(cartridge_id)
        for domain, score in domains:
            self.insert({
                "cartridge_id": cartridge_id,
                "domain": domain,
                "score": float(score),
            })
