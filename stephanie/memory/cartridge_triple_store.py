# stephanie/memory/cartridge_triple_store.py

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.models.cartridge_triple import CartridgeTripleORM


class CartridgeTripleStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "cartridge_triples"

    def insert(self, data: dict) -> CartridgeTripleORM:
        """
        Insert or update a triple for a cartridge.

        Expected dict keys: cartridge_id, subject, predicate, object, (optional) confidence
        """
        stmt = (
            pg_insert(CartridgeTripleORM)
            .values(**data)
            .on_conflict_do_nothing(index_elements=["cartridge_id", "subject", "predicate", "object"])
            .returning(CartridgeTripleORM.id)
        )

        result = self.session.execute(stmt)
        inserted_id = result.scalar()
        self.session.commit()

        if inserted_id and self.logger:
            self.logger.log("TripleUpserted", data)

        return (
            self.session.query(CartridgeTripleORM)
            .filter_by(
                cartridge_id=data["cartridge_id"],
                subject=data["subject"],
                predicate=data["predicate"],
                object=data["object"]
            )
            .first()
        )

    def get_triples(self, cartridge_id: int) -> list[CartridgeTripleORM]:
        return (
            self.session.query(CartridgeTripleORM)
            .filter_by(cartridge_id=cartridge_id)
            .all()
        )

    def delete_triples(self, cartridge_id: int):
        self.session.query(CartridgeTripleORM).filter_by(cartridge_id=cartridge_id).delete()
        self.session.commit()
        if self.logger:
            self.logger.log("TriplesDeleted", {"cartridge_id": cartridge_id})

    def set_triples(self, cartridge_id: int, triples: list[tuple[str, str, str, float]]):
        """
        Clear and re-add triples for a cartridge.
        Each triple is a tuple: (subject, predicate, object, confidence)
        """
        self.delete_triples(cartridge_id)
        for subj, pred, obj, conf in triples:
            self.insert({
                "cartridge_id": cartridge_id,
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "confidence": float(conf),
            })
